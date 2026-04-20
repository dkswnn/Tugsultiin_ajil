import { Controller, Get, Post, Body, Param, Put } from '@nestjs/common';
import type { CreateStudentDto } from './student.service';
import { StudentService } from './student.service';

@Controller('students')
export class StudentController {
  constructor(private readonly studentService: StudentService) {}

  @Post()
  async create(@Body() createStudentDto: CreateStudentDto) {
    return this.studentService.create(createStudentDto);
  }

  @Get()
  async findAll() {
    return this.studentService.findAll();
  }

  @Get(':studentId')
  async findById(@Param('studentId') studentId: string) {
    return this.studentService.findById(studentId);
  }

  @Put(':studentId')
  async update(
    @Param('studentId') studentId: string,
    @Body() updateData: Record<string, any>,
  ) {
    return this.studentService.update(studentId, updateData);
  }
}
